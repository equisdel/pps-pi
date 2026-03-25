package org.mybatis.jpetstore.mapper;

import org.mybatis.jpetstore.domain.Sequence;

/**
 * The Interface SequenceMapper.
 *
 * @author Eduardo Macarron
 */
public interface SequenceMapper {

  Sequence getSequence(Sequence sequence);

  void updateSequence(Sequence sequence);
}
